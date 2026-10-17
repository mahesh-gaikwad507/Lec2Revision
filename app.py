                use_container_width=True,
            )
            with st.expander("Translate this output"):
                target = st.selectbox("Translate to", ["Hindi", "Marathi", "English", "Gujarati", "Tamil", "Telugu", "Bengali", "French", "Spanish"], key="translate_target")
                if st.button("Translate output"):
                    with st.spinner("Translating..."):
                        try:
                            translated = translate_text(result, target)
                            st.session_state["translated_output"] = translated
                        except Exception as exc:
                            st.error(str(exc))
                if st.session_state.get("translated_output"):
                    st.markdown(st.session_state["translated_output"])
                    st.download_button("Download translation", st.session_state["translated_output"], file_name="translated_revision.md", mime="text/markdown")

    elif page == "Ask your lecture":
        st.subheader("💬 Ask questions about your learning material")
        source_text = st.session_state.get("source_text", "")
        if not source_text:
            st.info("First load source material on the Create revision material page.")
        else:
            st.caption(f"Current source: {st.session_state.get('source_name', 'Learning material')}")
            with db_connect() as conn:
                prior = conn.execute(
                    "SELECT role,message FROM chat_history WHERE user_id=? ORDER BY id DESC LIMIT 12",
                    (user_id,),
                ).fetchall()
            for item in reversed(prior):
                with st.chat_message("user" if item["role"] == "user" else "assistant"):
                    st.markdown(item["message"])
            question = st.chat_input("Ask a question about your lecture...")
            if question:
                subject = st.session_state.get("latest_subject", "")
                topic = st.session_state.get("latest_topic", "")
                save_chat(user_id, subject, topic, "user", question)
                context = build_context(question, source_text)
                prompt = (
                    f"Answer the student's question using the context below. If the answer is not in the context, "
                    f"say what is missing. Give a clear explanation and cite short quoted phrases from the context when useful.\n\n"
                    f"QUESTION: {question}\n\nCONTEXT:\n{context}"
                )
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            answer = ai_generate(prompt, "You are a patient tutor. Ground answers in the supplied lecture.", 0.2)
                            st.markdown(answer)
                            save_chat(user_id, subject, topic, "assistant", answer)
                        except Exception as exc:
                            st.error(str(exc))

    elif page == "My history":
        st.subheader("🗂️ Your generated revision history")
        rows = load_history(user_id)
        if not rows:
            st.info("No generated material yet. Create your first revision set.")
        else:
            for idx, row in enumerate(rows):
                label = f"{row['content_type']} — {row['subject'] or 'No subject'} / {row['topic'] or 'No topic'}"
                with st.expander(label):
                    st.caption(f"{row['created_at']} · Language: {row['language'] or 'English'}")
                    st.markdown(row["content"])
                    st.download_button(
                        "Download",
                        data=row["content"],
                        file_name=f"revision_{idx + 1}.md",
                        mime="text/markdown",
                        key=f"history_download_{idx}",
                    )

    elif page == "Dashboard":
        st.subheader("📈 Learning dashboard")
        rows = load_history(user_id, limit=1000)
        total = len(rows)
        subjects = len({row["subject"] for row in rows if row["subject"]})
        topics = len({row["topic"] for row in rows if row["topic"]})
        a, b, c = st.columns(3)
        a.metric("Generated resources", total)
        b.metric("Subjects explored", subjects)
        c.metric("Topics revised", topics)
        if rows:
            st.markdown("#### Recent activity")
            for row in rows[:10]:
                st.markdown(
                    f"- **{row['content_type']}** — {row['subject'] or 'Subject'} / "
                    f"{row['topic'] or 'Topic'} · {row['created_at']}"
                )
        else:
            st.info("Your dashboard will populate after you generate revision material.")


def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if st.session_state["authenticated"]:
        main_app()
    else:
        auth_screen()
    st.markdown(
        "<p class='small-note' style='text-align:center'>Lecture-to-Revision AI · Keep your API keys private.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
